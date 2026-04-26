"""CSV/Parquet cache for raw data requests.

Why: tushare has daily API quotas; we don't want to hammer it every run.
Every fetcher call funnels through this cache.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from utils.config import get_settings, resolve_path
from utils.logger import get_logger

log = get_logger(__name__)


def _cache_dir() -> Path:
    d = resolve_path(get_settings()["storage"]["cache_dir"])
    d.mkdir(parents=True, exist_ok=True)
    return d


def _market_dir() -> Path:
    d = resolve_path(get_settings()["storage"]["market_dir"])
    d.mkdir(parents=True, exist_ok=True)
    return d


def cache_key(api: str, **kwargs: Any) -> str:
    payload = json.dumps({"api": api, **kwargs}, sort_keys=True, default=str)
    return hashlib.md5(payload.encode("utf-8")).hexdigest()[:16]


def load_csv_cache(key: str) -> Optional[pd.DataFrame]:
    p = _cache_dir() / f"{key}.csv"
    if p.exists():
        if p.stat().st_size == 0:
            log.debug("cache EMPTY (skip) %s", p.name)
            return None
        try:
            df = pd.read_csv(p)
            if df.empty:
                return None
            log.debug("cache HIT  %s", p.name)
            return df
        except Exception:
            log.debug("cache CORRUPT (skip) %s", p.name)
            p.unlink(missing_ok=True)
            return None
    log.debug("cache MISS %s", p.name)
    return None


def save_csv_cache(key: str, df: pd.DataFrame) -> Path:
    p = _cache_dir() / f"{key}.csv"
    df.to_csv(p, index=False)
    return p


def save_parquet(ts_code: str, df: pd.DataFrame) -> Path:
    """Save one stock's OHLCV as `<ts_code>.parquet`."""
    p = _market_dir() / f"{ts_code}.parquet"
    df.to_parquet(p, index=False)
    return p


def load_parquet(ts_code: str) -> Optional[pd.DataFrame]:
    p = _market_dir() / f"{ts_code}.parquet"
    if not p.exists():
        return None
    return pd.read_parquet(p)


def list_cached_parquets() -> list[str]:
    """Return list of ts_code values that have a local parquet file."""
    return [p.stem for p in _market_dir().glob("*.parquet")]
