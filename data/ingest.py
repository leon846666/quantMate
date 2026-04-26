"""Bulk ingest: Tushare → PostgreSQL + Parquet.

Usage (as a script):
    python -m data.ingest --start 2018-01-01 --end 2022-12-31

Or import and call `ingest_all(...)` from code / main.py.
"""

from __future__ import annotations

import argparse
from typing import Iterable, Optional

import pandas as pd
from tqdm import tqdm

from data import fetcher, cache, database
from utils.logger import get_logger

log = get_logger(__name__)


def ingest_stock_basic() -> int:
    log.info("Ingesting stock_basic…")
    df = fetcher.fetch_stock_basic()
    if df.empty:
        log.warning("stock_basic is empty")
        return 0
    # normalise
    df = df.rename(columns={}).copy()
    for c in ("industry", "market", "list_date"):
        if c not in df.columns:
            df[c] = None
    df = df[["ts_code", "symbol", "name", "industry", "market", "list_date"]]
    df["list_date"] = pd.to_datetime(df["list_date"], errors="coerce")
    return database.upsert_df(df, "stock_basic", pk_cols=["ts_code"])


def ingest_daily_bars(
    ts_codes: Iterable[str],
    start: str,
    end: str,
    adj: str = "qfq",
    write_pg: bool = True,
) -> int:
    """Fetch OHLCV for each stock, save to Parquet and (optionally) PostgreSQL."""
    codes = list(ts_codes)
    total = 0
    for code in tqdm(codes, desc="daily OHLCV"):
        df = fetcher.fetch_daily(code, start, end, adj=adj)
        if df.empty:
            log.warning("No data returned for %s", code)
            continue
        # 1) Parquet
        cache.save_parquet(code, df)
        # 2) PostgreSQL daily_ohlcv
        if write_pg:
            pg_df = df.copy()
            pg_df["adj"] = adj
            pg_df["trade_date"] = pd.to_datetime(pg_df["trade_date"]).dt.date
            if "currency" not in pg_df.columns:
                pg_df["currency"] = fetcher.infer_currency(code)
            database.upsert_df(
                pg_df[["ts_code","trade_date","open","high","low","close","vol","amount","adj","currency"]],
                table="daily_ohlcv",
                pk_cols=["ts_code","trade_date","adj"],
            )
        total += len(df)
    log.info("Daily OHLCV ingested: %d rows across %d stocks", total, len(codes))
    return total


def ingest_daily_basic(ts_codes: Iterable[str], start: str, end: str) -> int:
    frames = []
    for code in tqdm(list(ts_codes), desc="daily_basic"):
        df = fetcher.fetch_daily_basic(code, start, end)
        if not df.empty:
            frames.append(df)
    if not frames:
        return 0
    all_df = pd.concat(frames, ignore_index=True)
    return database.upsert_df(all_df, "daily_basic", pk_cols=["ts_code", "trade_date"])


def ingest_all(
    start: str,
    end: str,
    universe: Optional[list[str]] = None,
    limit: Optional[int] = None,
) -> dict:
    """One-shot ingest.  Returns a dict of row counts.

    - `universe`: explicit list of ts_codes. If None, use all from stock_basic.
    - `limit`: cap number of stocks (useful while testing).
    """
    ingest_stock_basic()
    basic = database.query("SELECT ts_code FROM stock_basic")
    all_codes = basic["ts_code"].tolist() if not basic.empty else []
    codes = universe or all_codes
    if limit is not None:
        codes = codes[:limit]

    out = {"stocks": len(codes)}
    out["daily_rows"] = ingest_daily_bars(codes, start, end)
    out["daily_basic_rows"] = ingest_daily_basic(codes, start, end)
    log.info("Ingest complete: %s", out)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    ingest_all(start=args.start, end=args.end, limit=args.limit)


if __name__ == "__main__":
    main()
