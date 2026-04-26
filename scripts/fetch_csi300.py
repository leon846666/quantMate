"""scripts/fetch_csi300.py — 批量拉取沪深300成分股历史数据并存入 PostgreSQL

用法:
    python scripts/fetch_csi300.py
    python scripts/fetch_csi300.py --start 2023-04-21 --end 2026-04-21
    python scripts/fetch_csi300.py --batch-size 20   # 每批20只，控制API速率
    python scripts/fetch_csi300.py --resume          # 跳过已有数据的股票
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# ── 强制 UTF-8 输出（解决 Windows GBK 乱码）────────────────────────
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except AttributeError:
        pass

import akshare as ak
import pandas as pd
from tqdm import tqdm

from data import database, fetcher, ingest
from utils.logger import get_logger

log = get_logger(__name__)

DEFAULT_START = "2023-04-21"
DEFAULT_END   = "2026-04-21"


def _symbol_to_tscode(sym: str) -> str:
    sym = str(sym).zfill(6)
    if sym.startswith(("60", "68", "90")):
        return f"{sym}.SH"
    if sym.startswith(("0", "3", "20")):
        return f"{sym}.SZ"
    return f"{sym}.BJ"


def fetch_csi300_constituents() -> pd.DataFrame:
    """从 akshare 获取沪深300成分股列表，返回 DataFrame(ts_code, name)。"""
    print("  从 akshare 获取沪深300成分股 …")
    df = ak.index_stock_cons(symbol="000300")
    # 列名：品种代码 / 品种名称 (或 code/name，按 akshare 版本可能不同)
    code_col = df.columns[0]
    name_col = df.columns[1]
    df = df.rename(columns={code_col: "symbol", name_col: "name"})
    df["ts_code"] = df["symbol"].map(_symbol_to_tscode)
    return df[["ts_code", "name"]].reset_index(drop=True)


def _get_existing_codes(start: str, end: str) -> set[str]:
    """查询 PostgreSQL 中已有完整数据的股票代码（以有数据的日期数量判断）。"""
    try:
        sql = f"""
            SELECT ts_code, COUNT(*) AS cnt
            FROM daily_ohlcv
            WHERE adj = 'qfq'
              AND trade_date BETWEEN '{start}' AND '{end}'
            GROUP BY ts_code
        """
        df = database.query(sql)
        if df.empty:
            return set()
        # 至少有 200 条记录视为已完整拉取（约 1 年交易日）
        return set(df[df["cnt"] >= 200]["ts_code"].tolist())
    except Exception as e:
        log.warning("查询已有数据失败: %s", e)
        return set()


def run(
    start: str,
    end: str,
    batch_size: int,
    resume: bool,
    write_pg: bool,
    delay: float,
) -> None:
    print(f"\n{'='*60}")
    print(f"  沪深300 批量数据拉取")
    print(f"  区间: {start} → {end}")
    print(f"  批次: {batch_size} 只/批   延迟: {delay:.1f}s/只")
    print(f"{'='*60}\n")

    # ── 1. 获取成分股列表
    constituents = fetch_csi300_constituents()
    print(f"  沪深300 成分股: {len(constituents)} 只\n")

    # ── 2. 可选：跳过已有数据
    codes_to_fetch = constituents["ts_code"].tolist()
    if resume:
        existing = _get_existing_codes(start, end)
        codes_to_fetch = [c for c in codes_to_fetch if c not in existing]
        print(f"  已有数据: {len(existing)} 只  待拉取: {len(codes_to_fetch)} 只\n")

    if not codes_to_fetch:
        print("  所有股票均已有数据，无需重新拉取。")
        return

    # ── 3. 确保 DB 表存在，并写入股票名称
    if write_pg:
        schema_path = ROOT / "data" / "db" / "schema.sql"
        ddl = schema_path.read_text(encoding="utf-8")
        for stmt in [s.strip() for s in ddl.split(";") if s.strip()]:
            if not stmt.lstrip().startswith("--"):
                try:
                    database.execute(stmt)
                except Exception:
                    pass

        # 将成分股名称写入 stock_basic
        print("  写入 stock_basic (股票名称) …")
        basic_df = constituents.copy()
        basic_df["symbol"] = basic_df["ts_code"].str.split(".").str[0]
        basic_df["industry"] = None
        basic_df["market"] = basic_df["ts_code"].str.split(".").str[1]
        basic_df["list_date"] = None
        database.upsert_df(
            basic_df[["ts_code","symbol","name","industry","market","list_date"]],
            table="stock_basic",
            pk_cols=["ts_code"],
        )
        print(f"  stock_basic: {len(basic_df)} 条记录已 upsert")

    # ── 4. 分批拉取
    total_rows = 0
    failed: list[str] = []

    for i in range(0, len(codes_to_fetch), batch_size):
        batch = codes_to_fetch[i : i + batch_size]
        batch_num = i // batch_size + 1
        total_batches = (len(codes_to_fetch) + batch_size - 1) // batch_size
        print(f"[批次 {batch_num}/{total_batches}] 拉取 {len(batch)} 只 …")

        for code in tqdm(batch, desc=f"  batch-{batch_num}"):
            try:
                df = fetcher.fetch_daily(code, start, end, adj="qfq")
                if df.empty:
                    log.warning("无数据: %s", code)
                    failed.append(code)
                    continue

                # 写 Parquet
                from data import cache as data_cache
                data_cache.save_parquet(code, df)

                # 写 PostgreSQL
                if write_pg:
                    pg_df = df.copy()
                    pg_df["adj"] = "qfq"
                    pg_df["trade_date"] = pd.to_datetime(pg_df["trade_date"]).dt.date
                    if "currency" not in pg_df.columns:
                        pg_df["currency"] = fetcher.infer_currency(code)
                    database.upsert_df(
                        pg_df[["ts_code","trade_date","open","high","low",
                               "close","vol","amount","adj","currency"]],
                        table="daily_ohlcv",
                        pk_cols=["ts_code","trade_date","adj"],
                    )
                total_rows += len(df)

                if delay > 0:
                    time.sleep(delay)

            except Exception as e:
                log.error("拉取失败 %s: %s", code, e)
                failed.append(code)

        # 批次间稍长停顿，避免触发 API 限频
        if i + batch_size < len(codes_to_fetch):
            time.sleep(2)

    # ── 5. 汇总
    print(f"\n{'='*60}")
    print(f"  完成！")
    print(f"  总行数   : {total_rows:,}")
    print(f"  成功只数 : {len(codes_to_fetch) - len(failed)}")
    print(f"  失败只数 : {len(failed)}")
    if failed:
        print(f"  失败列表 : {failed[:20]}{'...' if len(failed) > 20 else ''}")
    print(f"{'='*60}\n")

    # ── 6. 快速预览 DB 记录数
    try:
        cnt = database.query(
            f"SELECT COUNT(*) AS n FROM daily_ohlcv "
            f"WHERE adj='qfq' AND trade_date BETWEEN '{start}' AND '{end}'"
        )
        print(f"  数据库 daily_ohlcv 共 {cnt.iloc[0,0]:,} 条（{start}～{end}，qfq）")
    except Exception:
        pass


def main() -> None:
    ap = argparse.ArgumentParser(description="批量拉取沪深300成分股 OHLCV")
    ap.add_argument("--start",      default=DEFAULT_START)
    ap.add_argument("--end",        default=DEFAULT_END)
    ap.add_argument("--batch-size", type=int, default=20,
                    help="每批只数（默认20）")
    ap.add_argument("--delay",      type=float, default=0.3,
                    help="每只股票请求间隔秒数（默认0.3s，防限频）")
    ap.add_argument("--resume",     action="store_true",
                    help="跳过已有 ≥200 条数据的股票")
    ap.add_argument("--no-pg",      dest="no_pg", action="store_true",
                    help="只写 Parquet，不写 PostgreSQL")
    args = ap.parse_args()

    run(
        start=args.start,
        end=args.end,
        batch_size=args.batch_size,
        resume=args.resume,
        write_pg=not args.no_pg,
        delay=args.delay,
    )


if __name__ == "__main__":
    main()
