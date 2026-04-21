"""Performance metrics: total return, CAGR, max drawdown, Sharpe, IC/IR."""
from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def daily_returns(nav: pd.Series) -> pd.Series:
    return nav.pct_change().dropna()


def total_return(nav: pd.Series) -> float:
    if nav.empty:
        return float("nan")
    return nav.iloc[-1] / nav.iloc[0] - 1


def cagr(nav: pd.Series) -> float:
    if nav.empty or len(nav) < 2:
        return float("nan")
    years = (nav.index[-1] - nav.index[0]).days / 365.25
    if years <= 0:
        return float("nan")
    return (nav.iloc[-1] / nav.iloc[0]) ** (1 / years) - 1


def annualised_volatility(rets: pd.Series) -> float:
    if rets.empty:
        return float("nan")
    return rets.std() * np.sqrt(TRADING_DAYS)


def sharpe(rets: pd.Series, rf: float = 0.0) -> float:
    if rets.empty or rets.std() == 0:
        return float("nan")
    excess = rets - rf / TRADING_DAYS
    return excess.mean() / excess.std() * np.sqrt(TRADING_DAYS)


def max_drawdown(nav: pd.Series) -> float:
    if nav.empty:
        return float("nan")
    roll_max = nav.cummax()
    dd = nav / roll_max - 1
    return dd.min()


def win_rate(rets: pd.Series) -> float:
    if rets.empty:
        return float("nan")
    return (rets > 0).mean()


def calmar(nav: pd.Series) -> float:
    mdd = max_drawdown(nav)
    if mdd == 0 or np.isnan(mdd):
        return float("nan")
    return cagr(nav) / abs(mdd)


def summary(nav: pd.Series) -> dict:
    nav = nav.dropna()
    rets = daily_returns(nav)
    return {
        "start": str(nav.index[0].date()) if not nav.empty else None,
        "end": str(nav.index[-1].date()) if not nav.empty else None,
        "days": len(nav),
        "total_return": total_return(nav),
        "cagr": cagr(nav),
        "ann_vol": annualised_volatility(rets),
        "sharpe": sharpe(rets),
        "max_drawdown": max_drawdown(nav),
        "calmar": calmar(nav),
        "win_rate": win_rate(rets),
    }


# -------- factor / prediction-quality metrics --------

def ic(predictions: pd.DataFrame, realised: pd.DataFrame,
       method: str = "spearman") -> pd.Series:
    """Per-day Information Coefficient.

    Parameters
    ----------
    predictions : columns ['trade_date','ts_code','score']
    realised    : columns ['trade_date','ts_code','fwd_ret']
    method      : 'pearson' or 'spearman'

    Returns
    -------
    Series indexed by trade_date.
    """
    merged = predictions.merge(
        realised, on=["trade_date", "ts_code"], how="inner"
    )
    out = []
    for day, g in merged.groupby("trade_date"):
        if len(g) < 5:
            out.append((day, np.nan))
            continue
        out.append((day, g["score"].corr(g["fwd_ret"], method=method)))
    return pd.Series(dict(out)).sort_index()


def ir(ic_series: pd.Series) -> float:
    ic_series = ic_series.dropna()
    if ic_series.empty or ic_series.std() == 0:
        return float("nan")
    return ic_series.mean() / ic_series.std() * np.sqrt(TRADING_DAYS)
