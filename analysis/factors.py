"""Factor library.

A factor function takes a DataFrame of one stock's timeseries (sorted by date)
with columns like `open, high, low, close, vol, amount, pe, pb, ...` and
returns a Series (or DataFrame with one column) indexed the same way as the
input's `trade_date`.

The driver in `analysis/features.py` applies each factor stock-by-stock and
stitches the result into a long panel indexed by (trade_date, ts_code).

Design rules:
    1. Factor functions stay pure — no I/O.
    2. Handle NaN gracefully (return NaN where input is missing).
    3. Never reference future data — use only `.rolling()` / `.shift(+n)`,
       never `.shift(-n)` except in the LABEL builder.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from analysis._registry import register_factor


# ============================================================
# Momentum / reversal family
# ============================================================


@register_factor("mom_5")
def mom_5(df: pd.DataFrame) -> pd.Series:
    """5-day return"""
    return df["close"].pct_change(5)


@register_factor("mom_10")
def mom_10(df: pd.DataFrame) -> pd.Series:
    return df["close"].pct_change(10)


@register_factor("mom_20")
def mom_20(df: pd.DataFrame) -> pd.Series:
    return df["close"].pct_change(20)


@register_factor("mom_60")
def mom_60(df: pd.DataFrame) -> pd.Series:
    return df["close"].pct_change(60)


@register_factor("reversal_1")
def reversal_1(df: pd.DataFrame) -> pd.Series:
    """Short-term reversal: yesterday's return, negated."""
    return -df["close"].pct_change(1)


@register_factor("reversal_5")
def reversal_5(df: pd.DataFrame) -> pd.Series:
    return -df["close"].pct_change(5)


# ============================================================
# Volatility
# ============================================================


@register_factor("vol_20")
def vol_20(df: pd.DataFrame) -> pd.Series:
    return df["close"].pct_change().rolling(20).std()


@register_factor("vol_60")
def vol_60(df: pd.DataFrame) -> pd.Series:
    return df["close"].pct_change().rolling(60).std()


@register_factor("downside_vol_20")
def downside_vol_20(df: pd.DataFrame) -> pd.Series:
    r = df["close"].pct_change()
    neg = r.where(r < 0, 0.0)
    return neg.rolling(20).std()


# ============================================================
# Liquidity / volume
# ============================================================


@register_factor("turnover_5")
def turnover_5(df: pd.DataFrame) -> pd.Series:
    if "turnover_rate" not in df.columns:
        return pd.Series(np.nan, index=df.index)
    return df["turnover_rate"].rolling(5).mean()


@register_factor("turnover_20")
def turnover_20(df: pd.DataFrame) -> pd.Series:
    if "turnover_rate" not in df.columns:
        return pd.Series(np.nan, index=df.index)
    return df["turnover_rate"].rolling(20).mean()


@register_factor("amount_20")
def amount_20(df: pd.DataFrame) -> pd.Series:
    return df["amount"].rolling(20).mean()


@register_factor("volume_ratio_5_20")
def volume_ratio_5_20(df: pd.DataFrame) -> pd.Series:
    return df["vol"].rolling(5).mean() / df["vol"].rolling(20).mean()


# ============================================================
# Valuation
# ============================================================


@register_factor("ep")
def ep(df: pd.DataFrame) -> pd.Series:
    """Earnings yield = 1 / PE (more robust than PE)."""
    if "pe" not in df.columns:
        return pd.Series(np.nan, index=df.index)
    return 1.0 / df["pe"].replace(0, np.nan)


@register_factor("bp")
def bp(df: pd.DataFrame) -> pd.Series:
    """Book-to-price = 1 / PB."""
    if "pb" not in df.columns:
        return pd.Series(np.nan, index=df.index)
    return 1.0 / df["pb"].replace(0, np.nan)


@register_factor("sp")
def sp(df: pd.DataFrame) -> pd.Series:
    """Sales-to-price = 1 / PS."""
    if "ps" not in df.columns:
        return pd.Series(np.nan, index=df.index)
    return 1.0 / df["ps"].replace(0, np.nan)


@register_factor("dp")
def dp(df: pd.DataFrame) -> pd.Series:
    """Dividend yield."""
    if "dv_ratio" not in df.columns:
        return pd.Series(np.nan, index=df.index)
    return df["dv_ratio"]


# ============================================================
# Size
# ============================================================


@register_factor("log_mcap")
def log_mcap(df: pd.DataFrame) -> pd.Series:
    if "total_mv" not in df.columns:
        return pd.Series(np.nan, index=df.index)
    return np.log(df["total_mv"].replace(0, np.nan))


# ============================================================
# Technical
# ============================================================


@register_factor("price_to_ma20")
def price_to_ma20(df: pd.DataFrame) -> pd.Series:
    return df["close"] / df["close"].rolling(20).mean() - 1


@register_factor("price_to_ma60")
def price_to_ma60(df: pd.DataFrame) -> pd.Series:
    return df["close"] / df["close"].rolling(60).mean() - 1


@register_factor("rsi_14")
def rsi_14(df: pd.DataFrame, period: int = 14) -> pd.Series:
    delta = df["close"].diff()
    up = delta.clip(lower=0).rolling(period).mean()
    down = -delta.clip(upper=0).rolling(period).mean()
    rs = up / down.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


@register_factor("atr_14")
def atr_14(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(period).mean() / df["close"]


@register_factor("skew_20")
def skew_20(df: pd.DataFrame) -> pd.Series:
    return df["close"].pct_change().rolling(20).skew()


@register_factor("kurt_20")
def kurt_20(df: pd.DataFrame) -> pd.Series:
    return df["close"].pct_change().rolling(20).kurt()


# ============================================================
# Label (future return) — NOT a factor; used to build y
# ============================================================


def future_return(df: pd.DataFrame, horizon: int = 5) -> pd.Series:
    """Forward return — THIS USES FUTURE DATA INTENTIONALLY.

    Only call when building training labels, never for features at live time.
    """
    return df["close"].shift(-horizon) / df["close"] - 1
