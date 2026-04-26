"""scripts/fetch_watchlist.py — 拉取自选股 3 年历史数据并存入 PostgreSQL + Parquet.

用法:
    python scripts/fetch_watchlist.py
    python scripts/fetch_watchlist.py --start 2022-01-01 --end 2025-12-31
    python scripts/fetch_watchlist.py --no-pg   # 只写 Parquet，不入库
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# 确保能找到项目根目录
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# ── 强制 UTF-8 输出（解决 Windows GBK 乱码）────────────────────────
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except AttributeError:
        pass

import pandas as pd
from typing import cast
from tqdm import tqdm

from data import fetcher, cache, database, ingest
from utils.config import get_settings
from utils.logger import get_logger

log = get_logger(__name__)

# ------------------------------------------------------------------ #
# 自选股清单                                                           #
# ------------------------------------------------------------------ #
WATCHLIST: list[dict] = [
    {"ts_code": "603920.SH", "name": "世运电路"},
    {"ts_code": "300750.SZ", "name": "宁德时代"},
    {"ts_code": "300476.SZ", "name": "胜宏科技"},
    {"ts_code": "002594.SZ", "name": "比亚迪"},
]

DEFAULT_START = "2023-04-21"  # 3 年前（2026-04-21 - 3y）
DEFAULT_END = "2026-04-21"


# ------------------------------------------------------------------ #
# 确保 PostgreSQL 表存在                                               #
# ------------------------------------------------------------------ #
def _ensure_tables() -> None:
    schema_path = ROOT / "data" / "db" / "schema.sql"
    ddl = schema_path.read_text(encoding="utf-8")
    # 逐条执行（跳过空行和纯注释行）
    statements = [s.strip() for s in ddl.split(";") if s.strip()]
    try:
        for stmt in statements:
            if stmt.lstrip().startswith("--"):
                continue
            database.execute(stmt)
        log.info("Schema applied (tables created if not exist)")
    except Exception as e:
        log.warning("Could not apply schema (DB might be offline): %s", e)


# ------------------------------------------------------------------ #
# 主逻辑                                                               #
# ------------------------------------------------------------------ #
def run(start: str, end: str, write_pg: bool = True) -> None:
    codes = [s["ts_code"] for s in WATCHLIST]
    names = {s["ts_code"]: s["name"] for s in WATCHLIST}

    print(f"\n{'='*55}")
    print(f"  自选股数据拉取  {start} → {end}")
    print(f"{'='*55}")
    for s in WATCHLIST:
        print(f"  {s['ts_code']}  {s['name']}")
    print(f"{'='*55}\n")

    if write_pg:
        _ensure_tables()

    # ---------- 1. OHLCV ----------
    print("[1/2] 拉取日线 OHLCV …")
    ohlcv_rows = ingest.ingest_daily_bars(
        ts_codes=codes,
        start=start,
        end=end,
        adj="qfq",
        write_pg=write_pg,
    )

    # ---------- 2. daily_basic (仅 A 股) ----------
    print("[2/2] 拉取 daily_basic (PE/PB/换手率，仅 A 股) …")
    a_codes = [c for c in codes if not c.upper().endswith(".HK")]
    basic_rows = 0
    frames = []
    for code in tqdm(a_codes, desc="daily_basic"):
        df = fetcher.fetch_daily_basic(code, start, end)
        if not df.empty:
            frames.append(df)
            basic_rows += len(df)

    if frames and write_pg:
        all_basic = cast(pd.DataFrame, pd.concat(frames, ignore_index=True))
        all_basic["trade_date"] = pd.to_datetime(all_basic["trade_date"]).dt.date
        database.upsert_df(
            all_basic,
            table="daily_basic",
            pk_cols=["ts_code", "trade_date"],
        )

    # ---------- 3. moneyflow via akshare (仅 A 股，约最近 6 个月) ----------
    print("[3/3] 拉取资金流向 (akshare，仅 A 股) …")
    mf_rows = 0
    mf_frames = []
    for code in tqdm(a_codes, desc="moneyflow"):
        df = fetcher.fetch_moneyflow_akshare(code)
        if not df.empty:
            mf_frames.append(df)
            mf_rows += len(df)

    if mf_frames and write_pg:
        all_mf = cast(pd.DataFrame, pd.concat(mf_frames, ignore_index=True))
        all_mf["trade_date"] = pd.to_datetime(all_mf["trade_date"]).dt.date
        # 补齐 moneyflow 表所需但 akshare 未提供的列
        for col in (
            "buy_sm_vol",
            "buy_md_vol",
            "buy_lg_vol",
            "buy_elg_vol",
            "sell_sm_vol",
            "sell_md_vol",
            "sell_lg_vol",
            "sell_elg_vol",
            "net_mf_vol",
        ):
            if col not in all_mf.columns:
                all_mf[col] = None
        # akshare 金额列 → net_mf_amount
        if "net_mf_amount" not in all_mf.columns:
            all_mf["net_mf_amount"] = None
        database.upsert_df(
            all_mf[
                [
                    "ts_code",
                    "trade_date",
                    "buy_sm_vol",
                    "buy_md_vol",
                    "buy_lg_vol",
                    "buy_elg_vol",
                    "sell_sm_vol",
                    "sell_md_vol",
                    "sell_lg_vol",
                    "sell_elg_vol",
                    "net_mf_vol",
                    "net_mf_amount",
                ]
            ],
            table="moneyflow",
            pk_cols=["ts_code", "trade_date"],
        )

    # ---------- 汇总 ----------
    print(f"\n{'='*55}")
    print(f"  完成！")
    print(f"  OHLCV 行数     : {ohlcv_rows:,}")
    print(f"  basic 行数     : {basic_rows:,}")
    print(f"  moneyflow 行数 : {mf_rows:,}")
    if write_pg:
        print(f"  PostgreSQL     : daily_ohlcv + moneyflow 已 upsert")
    print(f"{'='*55}\n")

    # ---------- 快速预览 ----------
    print("资金流向最新 5 行（世运电路 603920.SH）：")
    try:
        from data.database import query as db_query

        preview = db_query(
            "SELECT trade_date, net_mf_amount FROM moneyflow "
            "WHERE ts_code='603920.SH' ORDER BY trade_date DESC LIMIT 5"
        )
        print(preview.to_string(index=False))
    except Exception:
        print("  (数据库查询失败)")


def main() -> None:
    ap = argparse.ArgumentParser(description="拉取自选股历史数据")
    ap.add_argument("--start", default=DEFAULT_START, help="起始日期 YYYY-MM-DD")
    ap.add_argument("--end", default=DEFAULT_END, help="结束日期 YYYY-MM-DD")
    ap.add_argument(
        "--no-pg",
        dest="no_pg",
        action="store_true",
        help="跳过 PostgreSQL，只写 Parquet/CSV 缓存",
    )
    args = ap.parse_args()
    run(start=args.start, end=args.end, write_pg=not args.no_pg)


if __name__ == "__main__":
    main()
