"""Data fetcher — abstracts away the data source.

Three providers:
    - tushare   (primary, needs token)
    - akshare   (free fallback, no token)
    - mock      (synthetic, for unit tests and `main.py --demo`)

Which one runs is decided by `config/settings.yaml -> data_source.primary`.

All fetcher functions return pandas DataFrames with canonical column names:

    daily OHLCV:
        ['ts_code','trade_date','open','high','low','close','vol','amount']
    daily_basic:
        ['ts_code','trade_date','close','turnover_rate','pe','pb','pe_ttm',
         'ps','ps_ttm','dv_ratio','total_mv','circ_mv']
    stock_basic:
        ['ts_code','symbol','name','industry','market','list_date']
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
import pandas as pd

from data import cache
from utils.config import get_settings
from utils.date_utils import to_tushare_str
from utils.logger import get_logger

log = get_logger(__name__)

# -------- provider init (lazy) --------
_tushare_pro = None
_akshare = None


def _get_tushare():
    global _tushare_pro
    if _tushare_pro is not None:
        return _tushare_pro
    try:
        import tushare as ts
    except ImportError as e:
        raise ImportError("tushare not installed. `pip install tushare`") from e
    token = os.environ.get("TUSHARE_TOKEN") or get_settings()["data_source"]["tushare"]["token"]
    if not token or token.startswith("YOUR_"):
        raise RuntimeError(
            "Tushare token not set — fill config/settings.yaml or $TUSHARE_TOKEN."
        )
    ts.set_token(token)
    _tushare_pro = ts.pro_api()
    return _tushare_pro


def _get_akshare():
    global _akshare
    if _akshare is not None:
        return _akshare
    try:
        import akshare as ak
    except ImportError as e:
        raise ImportError("akshare not installed. `pip install akshare`") from e
    _akshare = ak
    return _akshare


def _provider() -> str:
    return get_settings()["data_source"]["primary"].lower()


# ============================================================
# stock_basic
# ============================================================

def fetch_stock_basic() -> pd.DataFrame:
    provider = _provider()
    key = cache.cache_key("stock_basic", provider=provider)
    cached = cache.load_csv_cache(key)
    if cached is not None:
        return cached

    if provider == "tushare":
        pro = _get_tushare()
        df = pro.stock_basic(
            exchange="",
            list_status="L",
            fields="ts_code,symbol,name,industry,market,list_date",
        )
    elif provider == "akshare":
        ak = _get_akshare()
        raw = ak.stock_info_a_code_name()
        df = raw.rename(columns={"code": "symbol"})
        df["ts_code"] = df["symbol"].map(_symbol_to_tscode)
        df["industry"] = None
        df["market"] = df["ts_code"].str[-2:]
        df["list_date"] = None
    elif provider == "mock":
        df = _mock_stock_basic()
    else:
        raise ValueError(f"Unknown data source provider: {provider}")

    cache.save_csv_cache(key, df)
    return df


def _symbol_to_tscode(sym: str) -> str:
    sym = str(sym).zfill(6)
    if sym.startswith(("60", "68", "90")):
        return f"{sym}.SH"
    if sym.startswith(("0", "3", "20")):
        return f"{sym}.SZ"
    return f"{sym}.BJ"


# ============================================================
# daily OHLCV
# ============================================================

def fetch_daily(
    ts_code: str,
    start: str,
    end: str,
    adj: str = "qfq",
) -> pd.DataFrame:
    provider = _provider()
    key = cache.cache_key("daily", ts_code=ts_code, start=start, end=end, adj=adj, provider=provider)
    cached = cache.load_csv_cache(key)
    if cached is not None:
        cached["trade_date"] = pd.to_datetime(cached["trade_date"])
        return cached

    if provider == "tushare":
        pro = _get_tushare()
        import tushare as ts
        df = ts.pro_bar(
            ts_code=ts_code,
            start_date=to_tushare_str(start),
            end_date=to_tushare_str(end),
            adj=adj,
            freq="D",
            api=pro,
        )
        if df is None or df.empty:
            df = pd.DataFrame(columns=[
                "ts_code","trade_date","open","high","low","close","vol","amount"
            ])
        else:
            df["trade_date"] = pd.to_datetime(df["trade_date"])
            df = df.sort_values("trade_date").reset_index(drop=True)
    elif provider == "akshare":
        ak = _get_akshare()
        sym = ts_code.split(".")[0]
        raw = ak.stock_zh_a_hist(
            symbol=sym,
            period="daily",
            start_date=to_tushare_str(start),
            end_date=to_tushare_str(end),
            adjust="qfq",
        )
        if raw is None or raw.empty:
            df = pd.DataFrame(columns=[
                "ts_code","trade_date","open","high","low","close","vol","amount"
            ])
        else:
            df = raw.rename(columns={
                "日期": "trade_date", "开盘": "open", "收盘": "close",
                "最高": "high",       "最低": "low",   "成交量": "vol",
                "成交额": "amount",
            })
            df["ts_code"] = ts_code
            df["trade_date"] = pd.to_datetime(df["trade_date"])
            df = df[["ts_code","trade_date","open","high","low","close","vol","amount"]]
    elif provider == "mock":
        df = _mock_daily(ts_code, start, end)
    else:
        raise ValueError(f"Unknown data source provider: {provider}")

    cache.save_csv_cache(key, df)
    return df


# ============================================================
# daily_basic (PE / PB / turnover / mcap)
# ============================================================

def fetch_daily_basic(ts_code: str, start: str, end: str) -> pd.DataFrame:
    provider = _provider()
    key = cache.cache_key("daily_basic", ts_code=ts_code, start=start, end=end, provider=provider)
    cached = cache.load_csv_cache(key)
    if cached is not None:
        cached["trade_date"] = pd.to_datetime(cached["trade_date"])
        return cached

    if provider == "tushare":
        pro = _get_tushare()
        df = pro.daily_basic(
            ts_code=ts_code,
            start_date=to_tushare_str(start),
            end_date=to_tushare_str(end),
            fields=(
                "ts_code,trade_date,close,turnover_rate,pe,pe_ttm,pb,ps,ps_ttm,"
                "dv_ratio,total_share,float_share,total_mv,circ_mv"
            ),
        )
        if df is None or df.empty:
            df = pd.DataFrame()
        else:
            df["trade_date"] = pd.to_datetime(df["trade_date"])
            df = df.sort_values("trade_date").reset_index(drop=True)
    elif provider == "mock":
        df = _mock_daily_basic(ts_code, start, end)
    else:
        # akshare doesn't have a clean equivalent — fall back to mock fields
        log.warning("daily_basic not supported on %s — falling back to mock", provider)
        df = _mock_daily_basic(ts_code, start, end)

    cache.save_csv_cache(key, df)
    return df


# ============================================================
# Mock generators (used by --demo and tests)
# ============================================================

_MOCK_INDUSTRIES = [
    "银行", "食品饮料", "医药生物", "电子", "计算机", "新能源",
    "有色金属", "化工", "机械设备", "房地产",
]


def _mock_stock_basic(n: int = 60, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        sym = f"{100000 + i:06d}"
        ts_code = _symbol_to_tscode(sym)
        rows.append({
            "ts_code": ts_code,
            "symbol": sym,
            "name": f"股票{i:03d}",
            "industry": rng.choice(_MOCK_INDUSTRIES),
            "market": ts_code[-2:],
            "list_date": "2010-01-01",
        })
    return pd.DataFrame(rows)


def _mock_daily(ts_code: str, start: str, end: str) -> pd.DataFrame:
    """Geometric Brownian motion with per-stock drift."""
    dates = pd.bdate_range(start=start, end=end)
    if len(dates) == 0:
        return pd.DataFrame(columns=["ts_code","trade_date","open","high","low","close","vol","amount"])
    seed = abs(hash(ts_code)) % (2**32 - 1)
    rng = np.random.default_rng(seed)
    drift = rng.uniform(-0.0003, 0.0006)
    vol = rng.uniform(0.012, 0.03)
    rets = rng.normal(drift, vol, size=len(dates))
    close = 10.0 * np.exp(np.cumsum(rets))
    open_ = close * (1 + rng.normal(0, 0.003, len(dates)))
    high = np.maximum(close, open_) * (1 + np.abs(rng.normal(0, 0.004, len(dates))))
    low = np.minimum(close, open_) * (1 - np.abs(rng.normal(0, 0.004, len(dates))))
    vol_ = rng.uniform(1e5, 1e7, len(dates))
    amount = vol_ * close / 1000
    return pd.DataFrame({
        "ts_code": ts_code,
        "trade_date": dates,
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "vol": vol_,
        "amount": amount,
    })


def _mock_daily_basic(ts_code: str, start: str, end: str) -> pd.DataFrame:
    daily = _mock_daily(ts_code, start, end)
    if daily.empty:
        return pd.DataFrame()
    seed = abs(hash(ts_code + "basic")) % (2**32 - 1)
    rng = np.random.default_rng(seed)
    n = len(daily)
    pe = np.clip(rng.normal(25, 15, n), 3, 200)
    pb = np.clip(rng.normal(3, 2, n), 0.3, 30)
    ps = np.clip(rng.normal(4, 3, n), 0.3, 40)
    tr = np.clip(rng.normal(2.5, 2, n), 0.01, 30)
    mv = np.clip(rng.lognormal(mean=10, sigma=0.8, size=n), 5e8, 2e12)
    return pd.DataFrame({
        "ts_code": ts_code,
        "trade_date": daily["trade_date"],
        "close": daily["close"],
        "turnover_rate": tr,
        "pe": pe,
        "pe_ttm": pe * rng.uniform(0.8, 1.2, n),
        "pb": pb,
        "ps": ps,
        "ps_ttm": ps * rng.uniform(0.8, 1.2, n),
        "dv_ratio": np.clip(rng.normal(1.5, 1.0, n), 0, 10),
        "total_share": mv / daily["close"].values,
        "float_share": (mv * 0.6) / daily["close"].values,
        "total_mv": mv / 1e4,
        "circ_mv": mv * 0.6 / 1e4,
    })


# ============================================================
# High-level helper: load one stock's data for modelling
# ============================================================

def load_panel(
    ts_codes: list[str],
    start: str,
    end: str,
    include_basic: bool = True,
) -> pd.DataFrame:
    """Return long-form panel: one row per (trade_date, ts_code)."""
    bars = []
    for code in ts_codes:
        bar = fetch_daily(code, start, end)
        if bar.empty:
            continue
        if include_basic:
            basic = fetch_daily_basic(code, start, end)
            if not basic.empty:
                bar = bar.merge(basic.drop(columns=["close"]),
                                on=["ts_code", "trade_date"], how="left")
        bars.append(bar)
    if not bars:
        return pd.DataFrame()
    panel = pd.concat(bars, ignore_index=True)
    panel = panel.sort_values(["trade_date", "ts_code"]).reset_index(drop=True)
    return panel
