"""quantMate — unified CLI.

Sub-commands (mutually exclusive flags; pick one):

    python main.py --demo       # end-to-end with mock data (no tushare / no PG)
    python main.py --fetch      # pull real data from tushare → PG + parquet
    python main.py --train      # train LightGBM from DB / parquet
    python main.py --backtest   # run LightGBM strategy backtest + report
    python main.py --list       # list registered factors and strategies

Use --help for full flag list.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Project-local imports (this module is the entry point; PYTHONPATH = repo root)
sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils.logger import get_logger
from utils.config import get_settings, resolve_path
from utils.date_utils import split_train_valid_test

log = get_logger("main")


# ============================================================
# Sub-command: --demo  (mock end-to-end)
# ============================================================

def run_demo(args: argparse.Namespace) -> int:
    """Full pipeline on mock data — no tushare / no PostgreSQL needed."""
    # Force mock provider for this run regardless of settings
    import os
    os.environ["QUANTMATE_PROVIDER_OVERRIDE"] = "mock"
    # Monkey-patch the provider getter
    import data.fetcher as _f

    def _mock_provider():
        return "mock"
    _f._provider = _mock_provider

    from analysis.features import make_features, build_training_matrix
    from analysis._registry import list_factors
    from data.fetcher import fetch_stock_basic, load_panel
    from strategy.lightgbm_strategy import (
        train_lightgbm, LGBParams, LightGBMStrategy, save_booster,
    )
    from strategy.base import StrategyConfig
    from backtest.engine import run_backtest, run_group_backtest, BacktestConfig
    from backtest.simulator import ExecutionConfig
    from evaluation.report import build_report, reports_dir
    from evaluation.metrics import summary, ic, ir
    from visualization.charts import plot_nav, plot_group_nav, plot_ic_series
    from visualization.dashboard import render_html

    start = args.start or "2018-01-01"
    end = args.end or "2022-12-31"
    n_stocks = args.stocks or 60
    log.info("DEMO: %d stocks, %s .. %s", n_stocks, start, end)

    # --- 1. fake stock universe + OHLCV + daily_basic ---
    stocks = fetch_stock_basic().head(n_stocks)
    ts_codes = stocks["ts_code"].tolist()
    panel = load_panel(ts_codes, start, end, include_basic=True)
    panel = panel.merge(
        stocks[["ts_code", "industry"]], on="ts_code", how="left"
    )
    log.info("Panel built: %d rows, %d stocks, %d days",
             len(panel), panel["ts_code"].nunique(),
             panel["trade_date"].nunique())

    # --- 2. factors ---
    feat_panel, feature_cols = make_features(
        panel, neutralize_by=["industry"], add_ranks=True
    )
    log.info("Factors computed: %d features", len(feature_cols))

    # --- 3. build training matrix ---
    horizon = get_settings()["model"].get("label_horizon", 5)
    mat = build_training_matrix(feat_panel, feature_cols, horizon=horizon)

    # Time-based train / valid / test split (7/2/1 of dates)
    all_days = sorted(mat["trade_date"].unique())
    n = len(all_days)
    train_end = all_days[int(n * 0.7)]
    valid_end = all_days[int(n * 0.9)]
    tr_mask, va_mask, te_mask = split_train_valid_test(
        mat["trade_date"], train_end=str(pd.Timestamp(train_end).date()),
        valid_end=str(pd.Timestamp(valid_end).date()),
    )
    Xtr, ytr = mat.loc[tr_mask, feature_cols], mat.loc[tr_mask, "fwd_ret"]
    Xva, yva = mat.loc[va_mask, feature_cols], mat.loc[va_mask, "fwd_ret"]
    log.info("Train=%d  Valid=%d  Test=%d", len(Xtr), len(Xva), te_mask.sum())

    # --- 4. train LightGBM ---
    lgb_cfg = get_settings()["model"]["lightgbm"]
    params = LGBParams(
        objective=lgb_cfg["objective"], metric=lgb_cfg["metric"],
        num_leaves=lgb_cfg["num_leaves"], learning_rate=lgb_cfg["learning_rate"],
        feature_fraction=lgb_cfg["feature_fraction"],
        bagging_fraction=lgb_cfg["bagging_fraction"],
        bagging_freq=lgb_cfg["bagging_freq"],
        min_data_in_leaf=min(lgb_cfg["min_data_in_leaf"], max(50, len(Xtr) // 20)),
        num_boost_round=min(lgb_cfg["num_boost_round"], 200),
        early_stopping_rounds=lgb_cfg["early_stopping_rounds"],
    )
    booster = train_lightgbm(Xtr, ytr, Xva, yva, params)
    save_booster(booster, reports_dir() / "demo" / "lgb_booster.pkl")

    # --- 5. predictions for the full panel (used by backtest + IC) ---
    test_rows = feat_panel.loc[feat_panel["trade_date"] > valid_end].copy()
    test_rows = test_rows.dropna(subset=feature_cols)
    test_rows["score"] = booster.predict(test_rows[feature_cols])

    predictions = test_rows[["trade_date", "ts_code", "score"]]
    realised = mat.loc[te_mask, ["trade_date", "ts_code", "fwd_ret"]]

    # IC / IR
    ic_series = ic(predictions, realised)
    log.info("Out-of-sample IC mean=%.4f  std=%.4f  IR=%.3f",
             ic_series.mean(), ic_series.std(), ir(ic_series))

    # --- 6. backtest top-group (G9) long-only ---
    strat_cfg = StrategyConfig(top_n=min(10, n_stocks // 6), weight="equal")
    strategy = LightGBMStrategy(
        booster, feature_cols, config=strat_cfg,
        group_count=9, top_group_only=True,
    )
    targets = strategy.generate_targets(test_rows)

    bt_cfg = BacktestConfig(
        initial_cash=1_000_000.0,
        start_date=str(pd.Timestamp(valid_end).date()),
        end_date=end,
        rebalance_freq="W-FRI",
        execution=ExecutionConfig(commission_rate=3e-4, slippage_bps=5.0),
    )
    nav, trades = run_backtest(panel, targets, bt_cfg)
    log.info("Backtest done. %d trades, NAV %.0f -> %.0f",
             len(trades), bt_cfg.initial_cash, nav["nav"].iloc[-1] if not nav.empty else np.nan)

    # --- 7. group-by-group backtest ---
    group_navs = run_group_backtest(panel, test_rows[["trade_date", "ts_code", "score"]], bt_cfg, n_groups=9)

    # --- 8. report + charts ---
    report_dir = reports_dir() / "demo"
    report_dir.mkdir(parents=True, exist_ok=True)

    plot_nav(nav, report_dir / "nav.png", title="Demo — G9 long, mock data")
    plot_group_nav(group_navs, report_dir / "group_nav.png")
    plot_ic_series(ic_series, report_dir / "ic.png",
                   title=f"Daily IC (mean={ic_series.mean():.3f}, IR={ir(ic_series):.2f})")

    md = build_report("demo", nav, trades, extras={
        "ic_mean": float(ic_series.mean()),
        "ic_std": float(ic_series.std()),
        "ir": float(ir(ic_series)),
        "features": len(feature_cols),
        "train_rows": int(len(Xtr)),
        "valid_rows": int(len(Xva)),
        "test_rows": int(te_mask.sum()),
    })
    render_html(
        report_dir / "index.html",
        title="quantMate DEMO report",
        summary_md=md.read_text(encoding="utf-8"),
        images=[report_dir / "nav.png",
                report_dir / "group_nav.png",
                report_dir / "ic.png"],
    )
    log.info("DEMO finished. Open: %s", report_dir / "index.html")
    return 0


# ============================================================
# Sub-command: --fetch (real tushare -> PG + parquet)
# ============================================================

def run_fetch(args: argparse.Namespace) -> int:
    from data.ingest import ingest_all
    ingest_all(start=args.start, end=args.end, limit=args.limit)
    return 0


# ============================================================
# Sub-command: --list
# ============================================================

def run_list(args: argparse.Namespace) -> int:
    from analysis._registry import list_factors
    from strategy.registry import list_strategies
    print("Registered factors:")
    for f in list_factors():
        print(f"  - {f}")
    print("\nRegistered strategies:")
    for s in list_strategies():
        print(f"  - {s}")
    return 0


# ============================================================
# entrypoint
# ============================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="quantMate CLI")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--demo", action="store_true", help="End-to-end on mock data.")
    g.add_argument("--fetch", action="store_true", help="Pull data from Tushare into PG + parquet.")
    g.add_argument("--train", action="store_true", help="(Stub) Train LightGBM from DB.")
    g.add_argument("--backtest", action="store_true", help="(Stub) Run backtest from DB.")
    g.add_argument("--list", action="store_true", help="List registered factors and strategies.")

    p.add_argument("--start", default=None)
    p.add_argument("--end", default=None)
    p.add_argument("--stocks", type=int, default=None, help="Universe size (demo only).")
    p.add_argument("--limit", type=int, default=None, help="Cap #stocks during fetch.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.demo:
        return run_demo(args)
    if args.fetch:
        if not args.start or not args.end:
            log.error("--fetch requires --start and --end")
            return 2
        return run_fetch(args)
    if args.list:
        return run_list(args)
    if args.train or args.backtest:
        log.error("--train / --backtest are stubs in this version; use --demo or wire up ingest first.")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
